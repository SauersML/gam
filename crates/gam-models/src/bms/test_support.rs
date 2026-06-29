//! Runtime-dimension jet substrate for single-sourcing the BMS
//! Bernoulli-marginal-slope **flex** interior (#932).
//!
//! The flex interior derivative tower
//! ([`super::row_primary_hessian::…::compute_row_analytic_flex_from_parts_into`])
//! is the last hand-coded BMS derivative chain: it assembles the per-row
//! gradient and Hessian over the runtime primary count `r = 2 + |β_h| + |β_w|`
//! (marginal-η, slope, score-warp basis, link-deviation basis) from cell moments
//! via hand implicit-function-theorem formulas for the calibrated intercept
//! `a(θ)` (the `coeff_au` / `coeff_bu` / `g_au_fixed` / `g_bu_fixed` chains).
//!
//! #932 single-sources every family's derivative tower through ONE row
//! log-likelihood evaluated in a generic jet. The empirical-rigid BMS path
//! already does this over the const-generic `gam_math::jet_scalar::Order2<2>`
//! ([`super::cell_moment_assembly`]: `empirical_rigid_row_nll_jet`); the survival
//! flex path does it over a private runtime-dimension `FlexJet`
//! (`survival/marginal_slope/timepoint_exact/flex_jet.rs`). The BMS flex interior
//! has a **runtime** primary count, so it needs a runtime-dimension dense jet —
//! the const-generic `Order2<K>` cannot carry a runtime `r`, and the survival
//! `FlexJet` is private to its module.
//!
//! This module ports the proven runtime-dimension `FlexJet` / `Jet2` substrate
//! (value + dense `r`-gradient + `r×r`-Hessian, the order-≤2 truncation of the
//! Leibniz / Faà di Bruno rules — bit-identical channel-for-channel to
//! `gam_math::jet_tower::Tower2`, just `Vec`-backed) and the filtered
//! implicit-function-theorem lift that calibrates the intercept `a(θ)` directly
//! in the jet (the runtime-dimension analogue of
//! `gam_math::jet_scalar::filtered_implicit_solve_scalar`). The flex calibration
//! `F(a, θ) = −μ(m) + ∫ Φ(η(z; a, θ)) dν(z) = 0` is lifted with this operator,
//! exactly as the rigid path lifts `Σ_k π_k Φ(a + s·g·x_k) − μ(m) = 0`.
//!
//! Lives in the BMS **parent** module (a bare `#[cfg(test)] mod test_support;`
//! in `bms/mod.rs`) so both consumers — this module's own FD gates (below) and
//! the `bms::cell_moment_assembly` flex-fixture oracle gate — reach it as a
//! private child of their common ancestor `bms` via `crate::bms::test_support`
//! (`super::test_support` from within `bms`), with NO
//! `pub(crate)` on the module itself: the build.rs ban-scanner exempts a
//! `#[cfg(test)]` module ONLY when it is declared bare `mod NAME;` with an
//! allowed name (`test_support`), never `pub(crate) mod …`. The whole module is
//! `cfg(test)`, so the substrate carries no dead code into the shipped build.

/// Generic second-order jet over a runtime primary count, mirroring the survival
/// flex `FlexJet` trait. `compose_unary` is the Faà di Bruno composition
/// `f ∘ self` with the unary derivative stack `[f, f′, f″, f‴, f⁗]` at the value
/// channel; order-≤2 jets read only the first three.
pub(crate) trait RuntimeJet: Sized + Clone {
    fn value(&self) -> f64;
    fn add(&self, o: &Self) -> Self;
    fn sub(&self, o: &Self) -> Self;
    fn mul(&self, o: &Self) -> Self;
    fn scale(&self, s: f64) -> Self;
    fn compose_unary(&self, d: [f64; 5]) -> Self;
}

/// Value `v`, gradient `g[i]`, Hessian `h[i*p+j]` (row-major, symmetric) over a
/// runtime primary count `p = g.len()`. The order-≤2 truncation of the Leibniz /
/// Faà di Bruno rules — bit-identical to [`gam_math::jet_tower::Tower2`]
/// channel-for-channel, just `Vec`-backed.
#[derive(Clone, Debug)]
pub(crate) struct Jet2 {
    pub(crate) v: f64,
    pub(crate) g: Vec<f64>,
    pub(crate) h: Vec<f64>,
}

impl Jet2 {
    /// The constant jet at `x`: zero gradient, zero Hessian.
    pub(crate) fn constant(x: f64, p: usize) -> Self {
        Jet2 {
            v: x,
            g: vec![0.0; p],
            h: vec![0.0; p * p],
        }
    }

    /// The seeded primary `axis` at value `x`: unit gradient in slot `axis`, zero
    /// Hessian.
    pub(crate) fn primary(x: f64, axis: usize, p: usize) -> Self {
        let mut g = vec![0.0; p];
        if axis < p {
            g[axis] = 1.0;
        }
        Jet2 {
            v: x,
            g,
            h: vec![0.0; p * p],
        }
    }

    #[inline]
    pub(crate) fn p(&self) -> usize {
        self.g.len()
    }
}

impl RuntimeJet for Jet2 {
    #[inline]
    fn value(&self) -> f64 {
        self.v
    }
    fn add(&self, o: &Self) -> Self {
        let p = self.p();
        let mut g = vec![0.0; p];
        let mut h = vec![0.0; p * p];
        for i in 0..p {
            g[i] = self.g[i] + o.g[i];
        }
        for k in 0..p * p {
            h[k] = self.h[k] + o.h[k];
        }
        Jet2 {
            v: self.v + o.v,
            g,
            h,
        }
    }
    fn sub(&self, o: &Self) -> Self {
        let p = self.p();
        let mut g = vec![0.0; p];
        let mut h = vec![0.0; p * p];
        for i in 0..p {
            g[i] = self.g[i] - o.g[i];
        }
        for k in 0..p * p {
            h[k] = self.h[k] - o.h[k];
        }
        Jet2 {
            v: self.v - o.v,
            g,
            h,
        }
    }
    fn mul(&self, o: &Self) -> Self {
        let p = self.p();
        let mut g = vec![0.0; p];
        let mut h = vec![0.0; p * p];
        for i in 0..p {
            g[i] = self.v * o.g[i] + self.g[i] * o.v;
        }
        for i in 0..p {
            for j in 0..p {
                h[i * p + j] = self.v * o.h[i * p + j]
                    + self.g[i] * o.g[j]
                    + self.g[j] * o.g[i]
                    + self.h[i * p + j] * o.v;
            }
        }
        Jet2 {
            v: self.v * o.v,
            g,
            h,
        }
    }
    fn scale(&self, s: f64) -> Self {
        Jet2 {
            v: self.v * s,
            g: self.g.iter().map(|&x| x * s).collect(),
            h: self.h.iter().map(|&x| x * s).collect(),
        }
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        // Order-≤2 reads only [f, f', f''].
        let p = self.p();
        let (f, f1, f2) = (d[0], d[1], d[2]);
        let mut g = vec![0.0; p];
        let mut h = vec![0.0; p * p];
        for i in 0..p {
            g[i] = f1 * self.g[i];
        }
        for i in 0..p {
            for j in 0..p {
                h[i * p + j] = f2 * self.g[i] * self.g[j] + f1 * self.h[i * p + j];
            }
        }
        Jet2 { v: f, g, h }
    }
}

/// Lift the calibrated scalar root `a(θ)` of `F(a, θ) = 0` directly into the
/// runtime jet, given the converged primal root `a0` (so `value(F(a0)) = 0`) and
/// the inverse primal Jacobian `inv_fa = 1 / F_a(a0, θ0)`.
///
/// This is the runtime-dimension analogue of
/// [`gam_math::jet_scalar::filtered_implicit_solve_scalar`]: the filtered Hensel
/// / Newton iteration `a ← a − F(a)·inv_fa` is nilpotency-exact for an order-≤2
/// jet after `iters ≥ 2` passes — the value channel is fixed at the certified
/// root (`value(F) = 0`), and each pass lifts the gradient then the Hessian by
/// one nilpotency grade. The closure builds the constraint jet `F(a, θ)` from the
/// seeded primaries `θ`, with `a` carrying the lift's current derivative estimate.
pub(crate) fn filtered_implicit_solve_jet2(
    a0: f64,
    inv_fa: f64,
    iters: usize,
    p: usize,
    f: impl Fn(&Jet2) -> Jet2,
) -> Jet2 {
    let mut a = Jet2::constant(a0, p);
    for _ in 0..iters {
        let residual = f(&a);
        a = a.sub(&residual.scale(inv_fa));
    }
    a
}

/// Polynomial convolution `out[i+j] += a[i]·b[j]` over jet coefficients (the jet
/// image of multiplying two `z`-polynomials with jet coefficients).
/// `out.len() = a.len() + b.len() − 1`.
pub(crate) fn poly_conv_jet(a: &[Jet2], b: &[Jet2], p: usize) -> Vec<Jet2> {
    let mut out = vec![Jet2::constant(0.0, p); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            out[i + j] = out[i + j].add(&ai.mul(bj));
        }
    }
    out
}

/// Interior **base-moment jets** `Mₙ(θ) = ∫_{zL}^{zR} zⁿ·e^{−q(z;θ)} dz` over the
/// FIXED cell domain `[zL, zR]`, carried over the runtime jet. The latent weight
/// is `e^{−q}`, `q(z) = ½(z² + η(z)²)`, `η(z) = Σₖ cₖ·zᵏ` the cell cubic. Only `η`
/// (hence `q`) depends on θ, through the coefficient jets `cₖ(θ)`:
///
/// ```text
///   Mₙ(θ) = ∫ zⁿ e^{−q₀} · e^{−Δq} dz ,   Δq = q − q₀ = ½(η² − η₀²)
///         = η₀·δη + ½·δη² ,                δη = η − η₀ = Σₖ δcₖ·zᵏ  (value 0)
/// ```
///
/// `Δq` has zero value channel, so for an order-≤2 jet the exponential truncates
/// exactly: `e^{−Δq} = 1 − Δq + ½·Δq²` (every `Δqᵏ`, `k ≥ 3`, has zero value /
/// gradient / Hessian). Writing `e^{−Δq} = Σₘ Sₘ·zᵐ` with jet coefficients `Sₘ`,
/// the moment integral collapses onto the SCALAR base moments
/// `M⁰ₖ = ∫ zᵏ e^{−q₀} dz`:
///
/// ```text
///   Mₙ(θ) = Σₘ Sₘ · M⁰_{n+m} .
/// ```
///
/// `c_jets` are the cell coefficients `c₀..c₃` as jets of θ; `c0_scalar` their
/// base values; `scalar_moments` the base moments `M⁰ₖ`. `δη`/`η₀` are degree 3,
/// so `Δq` is degree 6 and `Δq²` degree 12 — `scalar_moments` must hold indices
/// up to `max_n + 12`. Returns `M₀..M_{max_n}` as jets. This is the FIXED-domain
/// interior piece; cells with a θ-moving edge (`z = (τ−a)/b` link crossings) add
/// a separate Leibniz sliver.
pub(crate) fn cell_base_moment_jets(
    c_jets: &[Jet2; 4],
    c0_scalar: [f64; 4],
    scalar_moments: &[f64],
    max_n: usize,
) -> Vec<Jet2> {
    let p = c_jets[0].p();
    let cst = |x: f64| Jet2::constant(x, p);
    // δcₖ = cₖ(θ) − cₖ⁰ (value 0), and η₀ as constant jets.
    let dc: [Jet2; 4] = std::array::from_fn(|k| c_jets[k].sub(&cst(c0_scalar[k])));
    let eta0: [Jet2; 4] = std::array::from_fn(|k| cst(c0_scalar[k]));
    // Δq = η₀·δη + ½·δη²  (degree-6 jet polynomial, 7 coefficients).
    let eta0_deta = poly_conv_jet(&eta0, &dc, p);
    let deta_sq = poly_conv_jet(&dc, &dc, p);
    let mut dq: Vec<Jet2> = (0..eta0_deta.len())
        .map(|m| eta0_deta[m].add(&deta_sq[m].scale(0.5)))
        .collect();
    // e^{−Δq} = 1 − Δq + ½·Δq²  (degree-12 jet polynomial S, 13 coefficients).
    let dq_sq = poly_conv_jet(&dq, &dq, p);
    // S starts as −Δq + ½Δq² (pad Δq up to the degree of Δq²), then +1 const.
    let s_len = dq_sq.len();
    dq.resize(s_len, cst(0.0));
    let mut s_poly: Vec<Jet2> = (0..s_len)
        .map(|m| dq[m].scale(-1.0).add(&dq_sq[m].scale(0.5)))
        .collect();
    s_poly[0] = s_poly[0].add(&cst(1.0));
    // Mₙ(θ) = Σₘ Sₘ · M⁰_{n+m}.
    (0..=max_n)
        .map(|n| {
            let mut acc = cst(0.0);
            for (m, s_m) in s_poly.iter().enumerate() {
                let mom = scalar_moments[n + m];
                acc = acc.add(&s_m.scale(mom));
            }
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Finite-difference a closure `theta -> value` along axis pairs, returning a
    /// dense `p×p` Hessian by central differences of central-difference
    /// gradients. Coarse (`h = 1e-4`), used only to confirm the jet channels have
    /// the right STRUCTURE; the bit-identity to `Tower2` is the exact gate.
    fn fd_grad_hess(theta: &[f64], step: f64, f: impl Fn(&[f64]) -> f64) -> (Vec<f64>, Vec<f64>) {
        let p = theta.len();
        let mut grad = vec![0.0; p];
        for i in 0..p {
            let mut tp = theta.to_vec();
            let mut tm = theta.to_vec();
            tp[i] += step;
            tm[i] -= step;
            grad[i] = (f(&tp) - f(&tm)) / (2.0 * step);
        }
        let mut hess = vec![0.0; p * p];
        for i in 0..p {
            for j in 0..p {
                let mut tpp = theta.to_vec();
                let mut tpm = theta.to_vec();
                let mut tmp = theta.to_vec();
                let mut tmm = theta.to_vec();
                tpp[i] += step;
                tpp[j] += step;
                tpm[i] += step;
                tpm[j] -= step;
                tmp[i] -= step;
                tmp[j] += step;
                tmm[i] -= step;
                tmm[j] -= step;
                hess[i * p + j] = (f(&tpp) - f(&tpm) - f(&tmp) + f(&tmm)) / (4.0 * step * step);
            }
        }
        (grad, hess)
    }

    /// Pins that the runtime-dimension `Jet2` algebra (add / sub / mul / scale /
    /// compose_unary) carries the order-≤2 channels correctly over a runtime `p`.
    #[test]
    fn runtime_jet2_algebra_matches_finite_differences_932() {
        let theta = [0.37_f64, -0.21, 0.84];
        let p = theta.len();

        let scalar = |t: &[f64]| -> f64 {
            let (x, y, z) = (t[0], t[1], t[2]);
            (x * y).exp() + (z * z + 1.0) * x
        };

        let x = Jet2::primary(theta[0], 0, p);
        let y = Jet2::primary(theta[1], 1, p);
        let z = Jet2::primary(theta[2], 2, p);
        let xy = x.mul(&y);
        let e = xy.value().exp();
        let exp_xy = xy.compose_unary([e, e, e, e, e]);
        let z2p1 = z.mul(&z).add(&Jet2::constant(1.0, p));
        let jet = exp_xy.add(&z2p1.mul(&x));

        let (fg, fh) = fd_grad_hess(&theta, 1e-4, scalar);

        assert!((jet.value() - scalar(&theta)).abs() < 1e-12);
        for i in 0..p {
            assert!(
                (jet.g[i] - fg[i]).abs() < 1e-6,
                "grad[{i}]: jet {} vs fd {}",
                jet.g[i],
                fg[i]
            );
            for j in 0..p {
                assert!(
                    (jet.h[i * p + j] - fh[i * p + j]).abs() < 1e-4,
                    "hess[{i},{j}]: jet {} vs fd {}",
                    jet.h[i * p + j],
                    fh[i * p + j]
                );
            }
        }
    }

    /// Pin the runtime-dimension implicit-function-theorem lift
    /// ([`filtered_implicit_solve_jet2`]) against the analytic IFT derivatives of
    /// a closed-form implicit function `F(a, θ) = a³ + θ₀·a + θ₁ = 0` — the very
    /// chain the hand BMS flex intercept formulas implement, here produced
    /// MECHANICALLY by the jet lift.
    #[test]
    fn runtime_jet2_implicit_lift_matches_analytic_ift_932() {
        let theta = [1.3_f64, -0.45];
        let p = theta.len();

        let mut a0 = 0.0_f64;
        for _ in 0..200 {
            let fa = a0 * a0 * a0 + theta[0] * a0 + theta[1];
            let dfa = 3.0 * a0 * a0 + theta[0];
            a0 -= fa / dfa;
        }
        let f_a = 3.0 * a0 * a0 + theta[0];
        let inv_fa = 1.0 / f_a;

        let theta0 = Jet2::primary(theta[0], 0, p);
        let theta1 = Jet2::primary(theta[1], 1, p);
        let constraint = |a: &Jet2| -> Jet2 {
            let a2 = a.mul(a);
            let a3 = a2.mul(a);
            a3.add(&theta0.mul(a)).add(&theta1)
        };
        let a_jet = filtered_implicit_solve_jet2(a0, inv_fa, 2, p, constraint);

        let a_t0 = -a0 / f_a;
        let a_t1 = -1.0 / f_a;
        let f_aa = 6.0 * a0;
        let a_t0t0 = -(2.0 * a_t0 + f_aa * a_t0 * a_t0) / f_a;
        let a_t0t1 = -(a_t1 + f_aa * a_t0 * a_t1) / f_a;
        let a_t1t1 = -(f_aa * a_t1 * a_t1) / f_a;

        assert!((a_jet.value() - a0).abs() < 1e-12);
        assert!((a_jet.g[0] - a_t0).abs() < 1e-12, "a_t0 {} vs {a_t0}", a_jet.g[0]);
        assert!((a_jet.g[1] - a_t1).abs() < 1e-12, "a_t1 {} vs {a_t1}", a_jet.g[1]);
        assert!(
            (a_jet.h[0] - a_t0t0).abs() < 1e-12,
            "a_t0t0 {} vs {a_t0t0}",
            a_jet.h[0]
        );
        assert!(
            (a_jet.h[1] - a_t0t1).abs() < 1e-12,
            "a_t0t1 {} vs {a_t0t1}",
            a_jet.h[1]
        );
        assert!(
            (a_jet.h[p + 1] - a_t1t1).abs() < 1e-12,
            "a_t1t1 {} vs {a_t1t1}",
            a_jet.h[p + 1]
        );
        assert!((a_jet.h[1] - a_jet.h[p]).abs() < 1e-14);
    }

    /// #932 Phase 2b GATE (cell branch, interior): the runtime-jet base moments
    /// `Mₙ(θ)` match a central difference of the EXACT kernel moment evaluator
    /// (`evaluate_cell_moments`) with the cell cubic coefficients `c₀..c₃` as the
    /// θ-axes, over the FIXED cell domain. Pins the `e^{−Δq}` interior expansion
    /// (the part the survival flex `base_moment_jets` carries) independent of the
    /// moving-edge sliver. A wrong Δq fold or a dropped `½Δq²` Hessian term would
    /// blow the bound by orders.
    #[test]
    fn cell_base_moment_jets_match_fd_932() {
        use crate::cubic_cell_kernel::{DenestedCubicCell, evaluate_cell_moments};

        let base = [0.10_f64, 0.50, -0.20, 0.10];
        let (left, right) = (-0.80_f64, 0.70_f64);
        let cell = |c: [f64; 4]| DenestedCubicCell {
            left,
            right,
            c0: c[0],
            c1: c[1],
            c2: c[2],
            c3: c[3],
        };
        let max_n = 4usize;
        // Δq is degree 6, Δq² degree 12 ⇒ need scalar moments up to max_n + 12.
        let scalar_deg = max_n + 12;
        let moments_at = |c: [f64; 4]| -> Vec<f64> {
            evaluate_cell_moments(cell(c), scalar_deg)
                .expect("cell moments")
                .moments
                .into_vec()
        };

        let p = 4usize;
        let c_jets: [Jet2; 4] = std::array::from_fn(|k| Jet2::primary(base[k], k, p));
        let scalar_moments = moments_at(base);
        let m_jets = cell_base_moment_jets(&c_jets, base, &scalar_moments, max_n);

        for n in 0..=max_n {
            assert!(
                (m_jets[n].v - scalar_moments[n]).abs() <= 1e-12 * scalar_moments[n].abs().max(1.0),
                "M[{n}] value {:+.12e} != scalar {:+.12e}",
                m_jets[n].v,
                scalar_moments[n]
            );
        }

        let h = 1e-4_f64;
        for n in 0..=max_n {
            for k in 0..p {
                let mut cp = base;
                let mut cm = base;
                cp[k] += h;
                cm[k] -= h;
                let fd_g = (moments_at(cp)[n] - moments_at(cm)[n]) / (2.0 * h);
                assert!(
                    (m_jets[n].g[k] - fd_g).abs() <= 1e-5 * fd_g.abs().max(1.0) + 1e-9,
                    "dM[{n}]/dc[{k}] jet {:+.12e} != fd {:+.12e}",
                    m_jets[n].g[k],
                    fd_g
                );
                for l in 0..p {
                    let mut cpp = base;
                    let mut cpm = base;
                    let mut cmp = base;
                    let mut cmm = base;
                    cpp[k] += h;
                    cpp[l] += h;
                    cpm[k] += h;
                    cpm[l] -= h;
                    cmp[k] -= h;
                    cmp[l] += h;
                    cmm[k] -= h;
                    cmm[l] -= h;
                    let fd_h = (moments_at(cpp)[n] - moments_at(cpm)[n] - moments_at(cmp)[n]
                        + moments_at(cmm)[n])
                        / (4.0 * h * h);
                    assert!(
                        (m_jets[n].h[k * p + l] - fd_h).abs() <= 1e-3 * fd_h.abs().max(1.0) + 1e-6,
                        "d2M[{n}]/dc[{k}]dc[{l}] jet {:+.12e} != fd {:+.12e}",
                        m_jets[n].h[k * p + l],
                        fd_h
                    );
                }
            }
        }
    }
}
