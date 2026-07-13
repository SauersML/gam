//! Runtime-dimension jet substrate for single-sourcing the BMS
//! Bernoulli-marginal-slope **flex** interior (#932).
//!
//! The flex interior derivative tower
//! ([`super::row_primary_hessian::…::lower_bms_flex_row_order2_from_parts`])
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

/// Leibniz **edge sliver** jet for one θ-moving cell edge: the order-≤2 jet of
/// `∫_{zE0}^{zE(θ)} zⁿ·e^{−q(z;θ)} dz`, the thin moving-boundary contribution a
/// θ-moving edge (`zE = (τ−a)/b` link crossing) adds on top of the fixed-domain
/// interior moments.
///
/// With `δ = zE(θ) − zE0` (value 0) the integral Taylor-expands in `δ`; for an
/// order-≤2 jet only two terms survive (`δᵏ`, `k ≥ 3`, has zero value/grad/Hess):
///
/// ```text
///   sliver = g_jet·δ + ½·g_z(zE0)·δ²
/// ```
///
/// `g_jet = zE0ⁿ·e^{−q(zE0;θ)}` is the integrand at the FIXED edge point carried
/// as a jet (its `e^{−Δq}` θ-motion supplies the `g′_θ ⊗ δ′` cross-Hessian — `g`
/// MUST be a jet); `g_z(zE0)` is the SCALAR `z`-derivative
/// `(n·zE0^{n−1} − zE0ⁿ·q_z)·e^{−q}`, `q_z = z + η·η′(z)` (it only rides `δ²`,
/// already second order, so the scalar value suffices). Verified to order 2
/// against a moving-domain quadrature (grad rel-err 7e-11, Hess 6e-7).
pub(crate) fn edge_sliver_jet(
    n: usize,
    c_jets: &[Jet2; 4],
    c0_scalar: [f64; 4],
    ze_jet: &Jet2,
    ze0: f64,
) -> Jet2 {
    let p = c_jets[0].p();
    let cst = |x: f64| Jet2::constant(x, p);
    // δη(zE0) = Σₖ δcₖ·zE0ᵏ  (jet, value 0).
    let mut deta = cst(0.0);
    let mut z_pow = 1.0_f64;
    for k in 0..4 {
        deta = deta.add(&c_jets[k].sub(&cst(c0_scalar[k])).scale(z_pow));
        z_pow *= ze0;
    }
    let eta0 = c0_scalar[0]
        + c0_scalar[1] * ze0
        + c0_scalar[2] * ze0 * ze0
        + c0_scalar[3] * ze0 * ze0 * ze0;
    // Δq(zE0) = η₀·δη + ½·δη²  (jet, value 0); e^{−Δq} = 1 − Δq + ½Δq².
    let dq = deta.scale(eta0).add(&deta.mul(&deta).scale(0.5));
    let edq = cst(1.0).sub(&dq).add(&dq.mul(&dq).scale(0.5));
    let q0 = 0.5 * (ze0 * ze0 + eta0 * eta0);
    let g0 = ze0.powi(n as i32) * (-q0).exp();
    let g_jet = edq.scale(g0); // zⁿ·e^{−q} at zE0, as a jet over θ.
    let delta = ze_jet.sub(&cst(ze0)); // value 0.
    // g_z(zE0) scalar: q_z = z + η·η′(z), η′ = c1 + 2c2 z + 3c3 z².
    let eta_p = c0_scalar[1] + 2.0 * c0_scalar[2] * ze0 + 3.0 * c0_scalar[3] * ze0 * ze0;
    let q_z = ze0 + eta0 * eta_p;
    let z_nm1 = if n == 0 {
        0.0
    } else {
        (n as f64) * ze0.powi(n as i32 - 1)
    };
    let g_z = (z_nm1 - ze0.powi(n as i32) * q_z) * (-q0).exp();
    g_jet.mul(&delta).add(&delta.mul(&delta).scale(0.5 * g_z))
}

/// Cell base-moment jets over a **θ-moving** domain `[zL(θ), zR(θ)]`:
/// `Mₙ(θ) = interior_{[zL0,zR0]}(θ) + sliverₙ(zR) − sliverₙ(zL)`. The interior is
/// the fixed-domain [`cell_base_moment_jets`] (the `e^{−Δq}` expansion over the
/// base edges); each θ-moving edge adds its Leibniz [`edge_sliver_jet`]. Use this
/// for cells whose edges are link-knot crossings `z = (τ−a)/b` that move with the
/// (lifted) intercept and slope; pass `cell_base_moment_jets` directly for cells
/// with fixed (`±∞` / fixed-knot) edges.
pub(crate) fn cell_base_moment_jets_moving(
    c_jets: &[Jet2; 4],
    c0_scalar: [f64; 4],
    scalar_moments: &[f64],
    max_n: usize,
    zl_jet: &Jet2,
    zl0: f64,
    zr_jet: &Jet2,
    zr0: f64,
) -> Vec<Jet2> {
    let interior = cell_base_moment_jets(c_jets, c0_scalar, scalar_moments, max_n);
    (0..=max_n)
        .map(|n| {
            let s_r = edge_sliver_jet(n, c_jets, c0_scalar, zr_jet, zr0);
            let s_l = edge_sliver_jet(n, c_jets, c0_scalar, zl_jet, zl0);
            interior[n].add(&s_r).sub(&s_l)
        })
        .collect()
}

/// The cell cubic coefficients `c₀..c₃(θ)` as jets, from their bivariate
/// `(a, b)` Taylor composed with the (lifted) intercept jet `a_jet` and slope
/// jet `b_jet`. This is the `(a, b)` core of the hand `coeff_u`/`coeff_au`/
/// `coeff_bu` chains (the score-warp / link-deviation basis channels add to
/// `c_k` on top of this, through their own coefficient maps).
///
/// `c_k(a, b)` enters the kernel calibration; its first/second `(a, b)` partials
/// are the kernel's `denested_cell_coefficient_partials` / `_second_partials`.
/// With `da = a_jet − a0` and `db = b_jet − b0` (both value 0), the order-≤2 jet
/// of `c_k(a(θ), b(θ))` is EXACTLY the second-order bivariate Taylor
///
/// ```text
///   c_k = c0_k + dc_da_k·da + dc_db_k·db
///       + ½·dc_daa_k·da² + dc_dab_k·da·db + ½·dc_dbb_k·db² ,
/// ```
///
/// regardless of higher-than-2nd `(a, b)` structure in `c_k` (a Jet2 reads only
/// `c_k`'s derivatives up to order 2). It carries the lift's intercept Hessian
/// correctly: `da²`'s Hessian is `2·a_θ⊗a_θ` and `da`'s is `a_θθ`, so the `c_k`
/// Hessian reproduces the chain rule `dc_da·a_θθ + dc_daa·a_θ⊗a_θ` (+ the `b`
/// and cross terms). Verified standalone against a synthetic cubic `c(a,b)`
/// (grad 2e-8, Hess 1.4e-7).
pub(crate) struct CellCoeffAbPartials {
    /// Base cell coefficients `c₀..c₃` at `(a0, b0)`.
    pub(crate) c0: [f64; 4],
    /// `∂c/∂a`, `∂c/∂b` (kernel `denested_cell_coefficient_partials`).
    pub(crate) dc_da: [f64; 4],
    pub(crate) dc_db: [f64; 4],
    /// `∂²c/∂a²`, `∂²c/∂a∂b`, `∂²c/∂b²` (kernel `denested_cell_second_partials`).
    pub(crate) dc_daa: [f64; 4],
    pub(crate) dc_dab: [f64; 4],
    pub(crate) dc_dbb: [f64; 4],
}

pub(crate) fn cell_coeff_jet_ab(
    part: &CellCoeffAbPartials,
    a_jet: &Jet2,
    b_jet: &Jet2,
    a0: f64,
    b0: f64,
) -> [Jet2; 4] {
    let p = a_jet.p();
    let cst = |x: f64| Jet2::constant(x, p);
    let da = a_jet.sub(&cst(a0)); // value 0
    let db = b_jet.sub(&cst(b0)); // value 0
    let da2 = da.mul(&da);
    let dadb = da.mul(&db);
    let db2 = db.mul(&db);
    std::array::from_fn(|k| {
        cst(part.c0[k])
            .add(&da.scale(part.dc_da[k]))
            .add(&db.scale(part.dc_db[k]))
            .add(&da2.scale(0.5 * part.dc_daa[k]))
            .add(&dadb.scale(part.dc_dab[k]))
            .add(&db2.scale(0.5 * part.dc_dbb[k]))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_test_support::fd_checker::{
        numerical_gradient_central_diff, numerical_hessian_central_diff,
    };
    use ndarray::Array1;

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

        // Central-difference grad/Hessian of the same scalar closure via the
        // canonical FD harness (coarse `h = 1e-4`); used only to confirm the jet
        // channels have the right STRUCTURE — the bit-identity to `Tower2` is the
        // exact gate.
        let theta_arr = Array1::from_iter(theta.iter().copied());
        let fg = numerical_gradient_central_diff(
            |v: &Array1<f64>| scalar(v.as_slice().expect("contiguous theta")),
            &theta_arr,
            1e-4,
        );
        let fh = numerical_hessian_central_diff(
            |v: &Array1<f64>| scalar(v.as_slice().expect("contiguous theta")),
            &theta_arr,
            1e-4,
        );

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
                    (jet.h[i * p + j] - fh[[i, j]]).abs() < 1e-4,
                    "hess[{i},{j}]: jet {} vs fd {}",
                    jet.h[i * p + j],
                    fh[[i, j]]
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
        assert!(
            (a_jet.g[0] - a_t0).abs() < 1e-12,
            "a_t0 {} vs {a_t0}",
            a_jet.g[0]
        );
        assert!(
            (a_jet.g[1] - a_t1).abs() < 1e-12,
            "a_t1 {} vs {a_t1}",
            a_jet.g[1]
        );
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

    /// #932 Phase 2b GATE (cell branch, MOVING domain): the runtime-jet moving
    /// base moments `Mₙ(θ)` (fixed interior + Leibniz edge slivers) match a
    /// central difference of `evaluate_cell_moments` with BOTH the cell cubic
    /// coefficients `c₀..c₃` AND the cell edges `left`/`right` as θ-axes (6 axes).
    /// Pins the moving-boundary flux ([`edge_sliver_jet`]) and its cross-coupling
    /// with the interior `e^{−Δq}` motion (the `c × edge` Hessian block) at order
    /// 2 — a dropped or mis-signed sliver would blow the bound.
    #[test]
    fn cell_base_moment_jets_moving_match_fd_932() {
        use crate::cubic_cell_kernel::{DenestedCubicCell, evaluate_cell_moments};

        let base_c = [0.10_f64, 0.50, -0.20, 0.10];
        let (zl0, zr0) = (-0.80_f64, 0.70_f64);
        let max_n = 4usize;
        let scalar_deg = max_n + 12;
        // θ = [c₀, c₁, c₂, c₃, left, right] (6 axes).
        let p = 6usize;
        let base6 = [base_c[0], base_c[1], base_c[2], base_c[3], zl0, zr0];
        let cell_of = |t: [f64; 6]| DenestedCubicCell {
            left: t[4],
            right: t[5],
            c0: t[0],
            c1: t[1],
            c2: t[2],
            c3: t[3],
        };
        let moments_at = |t: [f64; 6]| -> Vec<f64> {
            evaluate_cell_moments(cell_of(t), scalar_deg)
                .expect("cell moments")
                .moments
                .into_vec()
        };

        // Interior scalar base moments over the FIXED base domain [zl0, zr0].
        let scalar_moments = moments_at(base6);
        let c_jets: [Jet2; 4] = std::array::from_fn(|k| Jet2::primary(base_c[k], k, p));
        let zl_jet = Jet2::primary(zl0, 4, p);
        let zr_jet = Jet2::primary(zr0, 5, p);
        let m_jets = cell_base_moment_jets_moving(
            &c_jets,
            base_c,
            &scalar_moments,
            max_n,
            &zl_jet,
            zl0,
            &zr_jet,
            zr0,
        );

        for n in 0..=max_n {
            assert!(
                (m_jets[n].v - scalar_moments[n]).abs() <= 1e-12 * scalar_moments[n].abs().max(1.0),
                "moving M[{n}] value {:+.12e} != scalar {:+.12e}",
                m_jets[n].v,
                scalar_moments[n]
            );
        }

        let h = 1e-4_f64;
        for n in 0..=max_n {
            for k in 0..p {
                let mut tp = base6;
                let mut tm = base6;
                tp[k] += h;
                tm[k] -= h;
                let fd_g = (moments_at(tp)[n] - moments_at(tm)[n]) / (2.0 * h);
                assert!(
                    (m_jets[n].g[k] - fd_g).abs() <= 1e-5 * fd_g.abs().max(1.0) + 1e-9,
                    "moving dM[{n}]/dθ[{k}] jet {:+.12e} != fd {:+.12e}",
                    m_jets[n].g[k],
                    fd_g
                );
                for l in 0..p {
                    let mut tpp = base6;
                    let mut tpm = base6;
                    let mut tmp = base6;
                    let mut tmm = base6;
                    tpp[k] += h;
                    tpp[l] += h;
                    tpm[k] += h;
                    tpm[l] -= h;
                    tmp[k] -= h;
                    tmp[l] += h;
                    tmm[k] -= h;
                    tmm[l] -= h;
                    let fd_h = (moments_at(tpp)[n] - moments_at(tpm)[n] - moments_at(tmp)[n]
                        + moments_at(tmm)[n])
                        / (4.0 * h * h);
                    assert!(
                        (m_jets[n].h[k * p + l] - fd_h).abs() <= 2e-3 * fd_h.abs().max(1.0) + 1e-6,
                        "moving d2M[{n}]/dθ[{k}]dθ[{l}] jet {:+.12e} != fd {:+.12e}",
                        m_jets[n].h[k * p + l],
                        fd_h
                    );
                }
            }
        }
    }

    /// #932 Phase 2b GATE (cell branch): the cell cubic coefficient jets
    /// `c₀..c₃(a, b)` ([`cell_coeff_jet_ab`]) match a central difference of the
    /// EXACT kernel `denested_cell_coefficients` with the intercept `a` and slope
    /// `b` as the θ-axes — pinning that the bivariate `(a, b)` Taylor reproduces
    /// the kernel coefficient map's value/gradient/Hessian.
    #[test]
    fn cell_coeff_jet_ab_match_fd_932() {
        use crate::cubic_cell_kernel::{
            LocalSpanCubic, denested_cell_coefficient_partials, denested_cell_coefficients,
            denested_cell_second_partials,
        };

        let score_span = LocalSpanCubic {
            left: -1.0,
            right: 1.0,
            c0: 0.10,
            c1: 0.30,
            c2: -0.10,
            c3: 0.05,
        };
        let link_span = LocalSpanCubic {
            left: -1.20,
            right: 0.90,
            c0: 0.20,
            c1: 0.25,
            c2: 0.08,
            c3: -0.03,
        };
        let (a0, b0) = (0.31_f64, 0.60_f64);

        let c0 = denested_cell_coefficients(score_span, link_span, a0, b0);
        let (dc_da, dc_db) = denested_cell_coefficient_partials(score_span, link_span, a0, b0);
        let (dc_daa, dc_dab, dc_dbb) = denested_cell_second_partials(score_span, link_span, a0, b0);
        let part = CellCoeffAbPartials {
            c0,
            dc_da,
            dc_db,
            dc_daa,
            dc_dab,
            dc_dbb,
        };

        let p = 2usize; // axes (a, b)
        let a_jet = Jet2::primary(a0, 0, p);
        let b_jet = Jet2::primary(b0, 1, p);
        let cj = cell_coeff_jet_ab(&part, &a_jet, &b_jet, a0, b0);

        let coeff_at = |a: f64, b: f64| denested_cell_coefficients(score_span, link_span, a, b);
        let h = 1e-5_f64;
        for k in 0..4 {
            assert!(
                (cj[k].v - c0[k]).abs() <= 1e-12 * c0[k].abs().max(1.0),
                "c[{k}] value {:+.12e} != {:+.12e}",
                cj[k].v,
                c0[k]
            );
            let g_a = (coeff_at(a0 + h, b0)[k] - coeff_at(a0 - h, b0)[k]) / (2.0 * h);
            let g_b = (coeff_at(a0, b0 + h)[k] - coeff_at(a0, b0 - h)[k]) / (2.0 * h);
            assert!(
                (cj[k].g[0] - g_a).abs() <= 1e-6 * g_a.abs().max(1.0) + 1e-9,
                "dc[{k}]/da jet {:+.12e} != fd {:+.12e}",
                cj[k].g[0],
                g_a
            );
            assert!(
                (cj[k].g[1] - g_b).abs() <= 1e-6 * g_b.abs().max(1.0) + 1e-9,
                "dc[{k}]/db jet {:+.12e} != fd {:+.12e}",
                cj[k].g[1],
                g_b
            );
            let h_aa = (coeff_at(a0 + h, b0)[k] - 2.0 * c0[k] + coeff_at(a0 - h, b0)[k]) / (h * h);
            let h_bb = (coeff_at(a0, b0 + h)[k] - 2.0 * c0[k] + coeff_at(a0, b0 - h)[k]) / (h * h);
            let h_ab = (coeff_at(a0 + h, b0 + h)[k]
                - coeff_at(a0 + h, b0 - h)[k]
                - coeff_at(a0 - h, b0 + h)[k]
                + coeff_at(a0 - h, b0 - h)[k])
                / (4.0 * h * h);
            assert!(
                (cj[k].h[0] - h_aa).abs() <= 1e-4 * h_aa.abs().max(1.0) + 1e-5,
                "d2c[{k}]/da2 jet {:+.12e} != fd {:+.12e}",
                cj[k].h[0],
                h_aa
            );
            assert!(
                (cj[k].h[3] - h_bb).abs() <= 1e-4 * h_bb.abs().max(1.0) + 1e-5,
                "d2c[{k}]/db2 jet {:+.12e} != fd {:+.12e}",
                cj[k].h[3],
                h_bb
            );
            assert!(
                (cj[k].h[1] - h_ab).abs() <= 1e-4 * h_ab.abs().max(1.0) + 1e-5,
                "d2c[{k}]/dadb jet {:+.12e} != fd {:+.12e}",
                cj[k].h[1],
                h_ab
            );
        }
    }
}
