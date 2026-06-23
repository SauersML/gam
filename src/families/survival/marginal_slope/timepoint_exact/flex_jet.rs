//! Single-source flex survival row NLL over a runtime-`K` jet algebra (#932,
//! doc §C/§D/Unifying).
//!
//! The flex marginal-slope row negative log-likelihood is
//! ```text
//! ℓ = w·[ logΦ(−η₀) − (1−d)·logΦ(−η₁)
//!         + d·½η₁² − d·logχ₁ + d·½q₁² + d·logD₁ − d·logqd₁ + d·ln2π ]
//! ```
//! (`flex_sensitivity.rs:105`). [`flex_row_nll`] writes this **once** over a
//! generic [`FlexJet`] scalar; instantiating it at [`Jet2`] yields value /
//! gradient / Hessian (replacing the hand grad/Hessian loops in
//! `flex_sensitivity.rs`), at [`Jet3`] yields the contracted third
//! `D_dir H[u,v]`, and at [`Jet4`] the contracted fourth — replacing the
//! hand probit-chain + quotient-rule assembly in
//! `gpu::cpu_oracle_third/fourth_contraction`. The directional / bidirectional
//! contraction "directions" fall out of the nilpotent ε / δ seeds of the timepoint
//! jets, exactly as the packed `Order2`/`OneSeed`/`TwoSeed` scalars do for
//! location-scale — but here over a **runtime** primary count `p` (the flex
//! primary count `4 + |h| + |w| + 1` is large and variable, so a `Vec`-backed
//! jet avoids the const-generic monomorphization blow-up the packed scalars would
//! incur).
//!
//! The timepoint quantities `η₀, η₁, χ₁, D₁` arrive as jets carrying their own
//! θ-derivatives (the `eta_u`/`eta_uv` packs from `first_full`, the directional
//! `*_dir` packs from `directional`, the bidirectional `*_uv_uv` packs from
//! `bidirectional`); `q₁`/`qd₁` are seeded as plain primaries. The single-source
//! probit derivative stack `surv_stack` and the `ln` stack carry the only special
//! functions (humans own primitive stability, the algebra owns combinatorics).

use super::*;
use crate::families::bms::signed_probit_neglog_derivatives_up_to_fourth;
use crate::families::jet_scalar::{filtered_implicit_solve_scalar, Order2};
use crate::families::jet_tower::Tower2;
use crate::families::survival::marginal_slope::gpu;
use crate::inference::probability::signed_probit_logcdf_and_mills_ratio;

/// #932 Item 1 (doc §B): lift the calibration intercept jet `a(θ)` — value /
/// gradient / Hessian — by `filtered_implicit_solve_scalar` over the calibration
/// constraint `F(a, θ) = 0`, instead of the hand IFT closed forms. `F`'s
/// `(a, θ)` jet channels ARE the already-computed calibration partials:
/// `F_a = D` (`d_check`), `F_{θi} = −f_u[i]`, `F_aa = f_aa`,
/// `F_{aθi} = d_u[i]` (= `∂D/∂θ_i`), `F_{θiθj} = −f_uv[i][j]`. The filtered
/// Newton step `A ← A − F(A)/F_a` (2 iterations at `Order2`, the nilpotency
/// order) returns `A.g = a_u`, `A.h = a_uv` — reproducing the hand IFT
/// `a_u = f_u/D`, `a_uv = (f_uv − d_u·a_u − d_u·a_u − f_aa·a_u·a_u)/D` term for
/// term, but from the recurrence rather than a memorised string (`jet_tower`
/// `implicit_solve` pins that equivalence at 1e-12). `O(K²)` per timepoint.
fn lift_intercept_order2<const K: usize>(
    d_check: f64,
    f_u: &[f64],
    f_uv: &[f64],
    f_aa: f64,
    d_u: &[f64],
    a0: f64,
) -> [[f64; K]; K] {
    let residual = |a: &Order2<K>| -> Order2<K> {
        let ag = a.g();
        let ah = a.h();
        let mut g = [0.0_f64; K];
        let mut h = [[0.0_f64; K]; K];
        for i in 0..K {
            g[i] = d_check * ag[i] - f_u[i];
        }
        for i in 0..K {
            for j in 0..K {
                h[i][j] = d_check * ah[i][j]
                    + f_aa * ag[i] * ag[j]
                    + d_u[i] * ag[j]
                    + d_u[j] * ag[i]
                    - f_uv[i * K + j];
            }
        }
        Order2(Tower2 { v: 0.0, g, h })
    };
    let a = filtered_implicit_solve_scalar::<K, Order2<K>>(a0, 1.0 / d_check, 2, residual);
    a.h()
}

/// The `[f64; 5]` Faà di Bruno stack of `g(η) = logΦ(−η)` at `η`.
///
/// With `N(m) = −logΦ(m)` and `(k1,k2,k3,k4) = N′…N⁗(m)` at `m = −η`
/// (`signed_probit_neglog_derivatives_up_to_fourth`), the chain rule on
/// `g(η) = −N(−η)` gives `g′ = k1`, `g″ = −k2`, `g‴ = k3`, `g⁗ = −k4`. This is
/// the entry/exit survival stack; composing the timepoint η-jet with it
/// reproduces the hand `entry_u1 = −entry_k1`, `entry_u2 = entry_k2`, … mapping
/// (`flex_sensitivity.rs`, `gpu::cpu_oracle_*`).
#[inline]
fn surv_stack(eta: f64) -> Result<[f64; 5], String> {
    let (logcdf, _) = signed_probit_logcdf_and_mills_ratio(-eta);
    let (k1, k2, k3, k4) = signed_probit_neglog_derivatives_up_to_fourth(-eta, 1.0)?;
    Ok([logcdf, k1, -k2, k3, -k4])
}

/// The `[f64; 5]` Faà di Bruno stack of `ln(x)`.
#[inline]
fn ln_stack(x: f64) -> [f64; 5] {
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    [x.ln(), inv, -inv2, 2.0 * inv2 * inv, -6.0 * inv2 * inv2]
}

/// A runtime-`K` truncated-Taylor scalar: the row loss is written once against
/// this interface and re-instantiated at [`Jet2`] / [`Jet3`] / [`Jet4`] for the
/// value/grad/Hessian, contracted-third, and contracted-fourth channels.
trait FlexJet: Sized + Clone {
    fn value(&self) -> f64;
    fn add(&self, o: &Self) -> Self;
    fn sub(&self, o: &Self) -> Self;
    fn mul(&self, o: &Self) -> Self;
    fn scale(&self, s: f64) -> Self;
    /// Faà di Bruno composition `f ∘ self` with stack `[f, f′, f″, f‴, f⁗]`.
    fn compose_unary(&self, d: [f64; 5]) -> Self;
    /// `ln(self)` via [`ln_stack`] at the value channel.
    #[inline]
    fn ln(&self) -> Self {
        self.compose_unary(ln_stack(self.value()))
    }
    /// `1/self` via the reciprocal Faà di Bruno stack at the value channel.
    #[inline]
    fn recip(&self) -> Self {
        let x = self.value();
        let inv = 1.0 / x;
        let inv2 = inv * inv;
        self.compose_unary([
            inv,
            -inv2,
            2.0 * inv2 * inv,
            -6.0 * inv2 * inv2,
            24.0 * inv2 * inv2 * inv,
        ])
    }
    /// `exp(self)` via the exponential stack at the value channel.
    #[inline]
    fn exp(&self) -> Self {
        let e = self.value().exp();
        self.compose_unary([e, e, e, e, e])
    }
    /// `self + c` for a scalar constant `c` (value-channel shift, derivatives
    /// unchanged) via the affine composition stack `[v+c, 1, 0, 0, 0]`.
    #[inline]
    fn add_const(&self, c: f64) -> Self {
        self.compose_unary([self.value() + c, 1.0, 0.0, 0.0, 0.0])
    }
}

// ── §B moment engine: the de-nested cell moments over a FlexJet ─────────────
//
// #932 Item 2 (doc §D). The per-cell moments `M_n = ∫_{z_L}^{z_R} z^n e^{−q(z)} dz`
// (sextic `q`, no closed antiderivative) satisfy the SAME raising recurrence the
// numeric `cubic_cell_kernel::reduce_sextic_moments` uses —
//   `M_{n+5} = (n·M_{n−1} − Σ_{j=0}^{4} d[j]·M_{n+j} − b_n) / d[5]`,
// with `d = q'(z)` coefficients (`sextic_qprime_coefficients`) and boundary term
// `b_n = z_R^n e^{−q(z_R)} − z_L^n e^{−q(z_L)}` — so it ports to ANY `FlexJet`
// scalar verbatim. Carrying the cell coefficients `c0..c3` and the (moving) edges
// `z_L,z_R` as jets propagates the moments' θ-derivatives mechanically: the
// `Σ d[j]·M_{n+j}` term is the interior coefficient sensitivity and the boundary
// term `b_n` is exactly the §D moving-boundary flux (its edge-jet derivatives are
// the Leibniz `[z^n e^{−q}·z_edge']` contributions the hand `directional` path
// assembles by hand). The base moments `M_0..M_4` (the normalization integrals)
// arrive as jets from the cell evaluator — those carry the only transcendental
// (erf/series) content; the algebra owns the rest.

/// `q'(z)` coefficient jets `[d0..d5]` for `q = ½(z² + η²)`, `η = c0+c1 z+c2 z²+
/// c3 z³`, over `FlexJet` cell-coefficient jets — the jet image of
/// [`crate::families::cubic_cell_kernel::sextic_qprime_coefficients`].
fn qprime_coeffs_jet<J: FlexJet>(c: &[J; 4]) -> [J; 6] {
    let (c0, c1, c2, c3) = (&c[0], &c[1], &c[2], &c[3]);
    // d0 = c0·c1
    let d0 = c0.mul(c1);
    // d1 = 1 + c1² + 2·c0·c2   (the leading `+z` of q' supplies the constant 1)
    let d1 = c1.mul(c1).add(&c0.mul(c2).scale(2.0)).add_const(1.0);
    // d2 = 3·c0·c3 + 3·c1·c2
    let d2 = c0.mul(c3).add(&c1.mul(c2)).scale(3.0);
    // d3 = 4·c1·c3 + 2·c2²
    let d3 = c1.mul(c3).scale(4.0).add(&c2.mul(c2).scale(2.0));
    // d4 = 5·c2·c3
    let d4 = c2.mul(c3).scale(5.0);
    // d5 = 3·c3²
    let d5 = c3.mul(c3).scale(3.0);
    [d0, d1, d2, d3, d4, d5]
}

/// `q(z) = ½(z² + η(z)²)` evaluated at an edge jet `z`, with `η` from the cell
/// coefficient jets — the exponent whose `e^{−q}` is the boundary weight.
fn cell_q_at_jet<J: FlexJet>(c: &[J; 4], z: &J) -> J {
    // η = c0 + c1 z + c2 z² + c3 z³  (Horner)
    let eta = c[3]
        .mul(z)
        .add(&c[2])
        .mul(z)
        .add(&c[1])
        .mul(z)
        .add(&c[0]);
    // ½(z² + η²)
    z.mul(z).add(&eta.mul(&eta)).scale(0.5)
}

/// One boundary term `z^n·e^{−q(z)}` at a (possibly infinite) moving edge jet.
/// An infinite edge contributes nothing (matching the numeric
/// `moment_boundary_term_with_powers` short-circuit).
fn boundary_edge_term_jet<J: FlexJet>(c: &[J; 4], z: &J, z_pow_n: &J, finite: bool) -> Option<J> {
    if !finite {
        return None;
    }
    let q = cell_q_at_jet(c, z);
    let w = q.scale(-1.0).exp();
    Some(z_pow_n.mul(&w))
}

/// The sextic moment recurrence over a `FlexJet`: given the cell coefficient
/// jets `c`, the moving edge jets `(z_left, z_right)` with their finiteness, and
/// the base moment jets `M_0..M_4`, return `M_0..M_max` as jets. Bit-faithful to
/// `reduce_sextic_moments` term for term, but every operation in the `FlexJet`
/// algebra so the moments carry their exact θ-derivatives.
fn cell_moment_recurrence_jet<J: FlexJet + Clone>(
    c: &[J; 4],
    z_left: &J,
    left_finite: bool,
    z_right: &J,
    right_finite: bool,
    base_m0_m4: &[J; 5],
    max_degree: usize,
) -> Vec<J> {
    let d = qprime_coeffs_jet(c);
    let inv_lead = d[5].recip();
    let mut moments: Vec<J> = base_m0_m4.iter().cloned().collect();
    if max_degree < 5 {
        moments.truncate(max_degree + 1);
        return moments;
    }
    // Rolling z^n at each edge (jets), starting at n = 0 (z^0 = 1 = z/z).
    let one_l = z_left.recip().mul(z_left);
    let one_r = z_right.recip().mul(z_right);
    let mut left_pow = one_l;
    let mut right_pow = one_r;
    for n in 0..=(max_degree - 5) {
        let b_left = boundary_edge_term_jet(c, z_left, &left_pow, left_finite);
        let b_right = boundary_edge_term_jet(c, z_right, &right_pow, right_finite);
        // b_n = right − left, missing edges contribute zero.
        let mut b_n = match (b_right, b_left) {
            (Some(r), Some(l)) => r.sub(&l),
            (Some(r), None) => r,
            (None, Some(l)) => l.scale(-1.0),
            (None, None) => moments[0].scale(0.0),
        };
        // numer = n·M_{n−1} − Σ_{j=0}^{4} d[j]·M_{n+j} − b_n
        let mut numer = if n == 0 {
            moments[0].scale(0.0)
        } else {
            moments[n - 1].scale(n as f64)
        };
        for j in 0..=4 {
            numer = numer.sub(&d[j].mul(&moments[n + j]));
        }
        numer = numer.sub(&b_n);
        moments.push(numer.mul(&inv_lead));
        // Roll powers: z^{n+1} = z^n · z.
        left_pow = if left_finite {
            left_pow.mul(z_left)
        } else {
            b_n.scale(0.0)
        };
        right_pow = if right_finite {
            right_pow.mul(z_right)
        } else {
            // reuse b_n as a zero-jet scratch source of the right `p`
            b_n = b_n.scale(0.0);
            b_n
        };
    }
    moments
}

/// The single-source flex row NLL **minus** the additive `w·d·ln2π` constant
/// (which the caller adds to the value channel — it has no derivative). Written
/// once over `FlexJet`; the instantiating scalar selects the channel.
#[inline]
fn flex_row_nll<J: FlexJet>(
    eta0: &J,
    eta1: &J,
    chi1: &J,
    d1: &J,
    q1: &J,
    qd1: &J,
    surv0: [f64; 5],
    surv1: [f64; 5],
    wi: f64,
    di: f64,
) -> J {
    let wd = wi * di;
    // w·logΦ(−η₀)
    let mut nll = eta0.compose_unary(surv0).scale(wi);
    // −w(1−d)·logΦ(−η₁)
    nll = nll.add(&eta1.compose_unary(surv1).scale(-wi * (1.0 - di)));
    // +w·d·½η₁²   (the −d·logφ(η₁) term, sans ½ln2π const)
    nll = nll.add(&eta1.mul(eta1).scale(0.5 * wd));
    // +w·d·½q₁²   (the −d·logφ(q₁) term, sans ½ln2π const)
    nll = nll.add(&q1.mul(q1).scale(0.5 * wd));
    // −w·d·logχ₁
    nll = nll.sub(&chi1.ln().scale(wd));
    // +w·d·logD₁
    nll = nll.add(&d1.ln().scale(wd));
    // −w·d·logqd₁
    nll = nll.sub(&qd1.ln().scale(wd));
    nll
}

// ── Jet2: value / gradient / Hessian (runtime K) ───────────────────────────

/// Value `v`, gradient `g[i]`, Hessian `h[i*p+j]` (row-major, symmetric) over a
/// runtime primary count `p = g.len()`. The order-≤2 truncation of the Leibniz /
/// Faà di Bruno rules — bit-identical to [`super::super::super::jet_tower::Tower2`]
/// channel-for-channel, just `Vec`-backed.
#[derive(Clone)]
struct Jet2 {
    v: f64,
    g: Vec<f64>,
    h: Vec<f64>,
}

impl Jet2 {
    /// A jet from explicit channels: `g` length `p`, `h` length `p*p` (or empty
    /// for the grad-only path, treated as the zero Hessian).
    fn from_parts(v: f64, g: &[f64], h: &[f64]) -> Self {
        let p = g.len();
        let hv = if h.is_empty() {
            vec![0.0; p * p]
        } else {
            assert_eq!(h.len(), p * p, "Jet2::from_parts Hessian length");
            h.to_vec()
        };
        Jet2 {
            v,
            g: g.to_vec(),
            h: hv,
        }
    }

    /// A jet from a gradient view and optional Hessian view (contiguity-safe:
    /// copies element-wise). `None` Hessian is the grad-only path.
    fn from_view(v: f64, g: ndarray::ArrayView1<'_, f64>, h: Option<ndarray::ArrayView2<'_, f64>>) -> Self {
        let p = g.len();
        let gv: Vec<f64> = g.iter().copied().collect();
        let hv = match h {
            Some(hm) => {
                let mut out = vec![0.0; p * p];
                for i in 0..p {
                    for j in 0..p {
                        out[i * p + j] = hm[[i, j]];
                    }
                }
                out
            }
            None => vec![0.0; p * p],
        };
        Jet2 { v, g: gv, h: hv }
    }

    /// The seeded primary `p_axis` at value `x`: unit gradient in slot `axis`,
    /// zero Hessian.
    fn primary(x: f64, axis: usize, p: usize) -> Self {
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
    fn p(&self) -> usize {
        self.g.len()
    }
}

impl FlexJet for Jet2 {
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

// ── Jet3: one-seed directional, contracted third (doc §A.2) ────────────────

/// An [`Jet2`] base plus one nilpotent ε (`ε² = 0`) holding another [`Jet2`].
/// After seeding the timepoint jets' ε-parts with their directional derivatives,
/// the ε-Hessian of the evaluated NLL is `Σ_c ℓ_{abc} dir_c = (D_dir H)[a][b]`.
#[derive(Clone)]
struct Jet3 {
    base: Jet2,
    eps: Jet2,
}

impl Jet3 {
    /// Seeded primary: base = `primary(x, axis)`, ε = constant `dir[axis]`.
    fn primary(x: f64, axis: usize, p: usize, dir_axis: f64) -> Self {
        Jet3 {
            base: Jet2::primary(x, axis, p),
            eps: Jet2::from_parts(dir_axis, &vec![0.0; p], &[]),
        }
    }
    /// The contracted-third channel `Σ_c ℓ_{abc} dir_c` (the ε-Hessian).
    fn contracted_third(&self) -> Vec<f64> {
        self.eps.h.clone()
    }
}

impl FlexJet for Jet3 {
    #[inline]
    fn value(&self) -> f64 {
        self.base.v
    }
    fn add(&self, o: &Self) -> Self {
        Jet3 {
            base: self.base.add(&o.base),
            eps: self.eps.add(&o.eps),
        }
    }
    fn sub(&self, o: &Self) -> Self {
        Jet3 {
            base: self.base.sub(&o.base),
            eps: self.eps.sub(&o.eps),
        }
    }
    fn mul(&self, o: &Self) -> Self {
        Jet3 {
            base: self.base.mul(&o.base),
            eps: self.base.mul(&o.eps).add(&self.eps.mul(&o.base)),
        }
    }
    fn scale(&self, s: f64) -> Self {
        Jet3 {
            base: self.base.scale(s),
            eps: self.eps.scale(s),
        }
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let base = self.base.compose_unary([d[0], d[1], d[2], d[3], d[4]]);
        // f'(base) as a Jet2 (consumes [f', f'', f''']).
        let fprime = self.base.compose_unary([d[1], d[2], d[3], d[4], d[4]]);
        let eps = fprime.mul(&self.eps);
        Jet3 { base, eps }
    }
}

// ── Jet4: two-seed, contracted fourth (doc §A.3) ───────────────────────────

/// An [`Jet2`] base plus ε, δ (`ε² = δ² = 0`, `εδ` retained) — four [`Jet2`]
/// parts. After seeding with both directions, the εδ-Hessian of the NLL is
/// `Σ_{cd} ℓ_{abcd} u_c v_d`.
#[derive(Clone)]
struct Jet4 {
    base: Jet2,
    eps: Jet2,
    del: Jet2,
    eps_del: Jet2,
}

impl Jet4 {
    fn primary(x: f64, axis: usize, p: usize, du: f64, dv: f64) -> Self {
        let zero = vec![0.0; p];
        Jet4 {
            base: Jet2::primary(x, axis, p),
            eps: Jet2::from_parts(du, &zero, &[]),
            del: Jet2::from_parts(dv, &zero, &[]),
            eps_del: Jet2::from_parts(0.0, &zero, &[]),
        }
    }
    fn contracted_fourth(&self) -> Vec<f64> {
        self.eps_del.h.clone()
    }
}

impl FlexJet for Jet4 {
    #[inline]
    fn value(&self) -> f64 {
        self.base.v
    }
    fn add(&self, o: &Self) -> Self {
        Jet4 {
            base: self.base.add(&o.base),
            eps: self.eps.add(&o.eps),
            del: self.del.add(&o.del),
            eps_del: self.eps_del.add(&o.eps_del),
        }
    }
    fn sub(&self, o: &Self) -> Self {
        Jet4 {
            base: self.base.sub(&o.base),
            eps: self.eps.sub(&o.eps),
            del: self.del.sub(&o.del),
            eps_del: self.eps_del.sub(&o.eps_del),
        }
    }
    fn mul(&self, o: &Self) -> Self {
        let base = self.base.mul(&o.base);
        let eps = self.base.mul(&o.eps).add(&self.eps.mul(&o.base));
        let del = self.base.mul(&o.del).add(&self.del.mul(&o.base));
        let eps_del = self
            .base
            .mul(&o.eps_del)
            .add(&self.eps.mul(&o.del))
            .add(&self.del.mul(&o.eps))
            .add(&self.eps_del.mul(&o.base));
        Jet4 {
            base,
            eps,
            del,
            eps_del,
        }
    }
    fn scale(&self, s: f64) -> Self {
        Jet4 {
            base: self.base.scale(s),
            eps: self.eps.scale(s),
            del: self.del.scale(s),
            eps_del: self.eps_del.scale(s),
        }
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let base = self.base.compose_unary([d[0], d[1], d[2], d[3], d[4]]);
        let fprime = self.base.compose_unary([d[1], d[2], d[3], d[4], d[4]]);
        let fsecond = self.base.compose_unary([d[2], d[3], d[4], d[4], d[4]]);
        let eps = fprime.mul(&self.eps);
        let del = fprime.mul(&self.del);
        let eps_del = fsecond
            .mul(&self.eps)
            .mul(&self.del)
            .add(&fprime.mul(&self.eps_del));
        Jet4 {
            base,
            eps,
            del,
            eps_del,
        }
    }
}

/// `Σ_i x[i]·y[i]` over equal-length slices.
#[inline]
fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
}

/// `out[i] = Σ_j m[i*p+j]·v[j]` for a row-major `p×p` matrix `m`.
fn mat_vec(m: &[f64], v: &[f64], p: usize) -> Vec<f64> {
    let mut out = vec![0.0; p];
    for i in 0..p {
        let mut acc = 0.0;
        for j in 0..p {
            acc += m[i * p + j] * v[j];
        }
        out[i] = acc;
    }
    out
}

/// `v1ᵀ m v2` for a row-major `p×p` matrix `m`.
fn quad_form(m: &[f64], v1: &[f64], v2: &[f64], p: usize) -> f64 {
    let mut acc = 0.0;
    for i in 0..p {
        let mi = &m[i * p..i * p + p];
        acc += v1[i] * dot(mi, v2);
    }
    acc
}

/// Order-≤2 jet channels (value, gradient view, optional Hessian view) for the
/// four flex row-NLL inputs (entry η, exit η, observed χ, observed d), bundled
/// so `flex_row_nll_value_grad_hess` stays under the argument-count gate.
pub(crate) struct FlexRowJet2Channels<'a> {
    pub eta0_v: f64,
    pub eta0_g: ndarray::ArrayView1<'a, f64>,
    pub eta0_h: Option<ndarray::ArrayView2<'a, f64>>,
    pub eta1_v: f64,
    pub eta1_g: ndarray::ArrayView1<'a, f64>,
    pub eta1_h: Option<ndarray::ArrayView2<'a, f64>>,
    pub chi1_v: f64,
    pub chi1_g: ndarray::ArrayView1<'a, f64>,
    pub chi1_h: Option<ndarray::ArrayView2<'a, f64>>,
    pub d1_v: f64,
    pub d1_g: ndarray::ArrayView1<'a, f64>,
    pub d1_h: Option<ndarray::ArrayView2<'a, f64>>,
}

/// Entry/exit base + directional timepoint packs for the contracted-third path,
/// bundled to keep `flex_row_nll_third_contracted` under the argument-count gate.
pub(crate) struct FlexThirdPacks<'a> {
    pub entry_base: &'a gpu::SurvivalFlexBlock10TimepointBase,
    pub exit_base: &'a gpu::SurvivalFlexBlock10TimepointBase,
    pub entry_ext: &'a gpu::SurvivalFlexBlock10TimepointDirectional,
    pub exit_ext: &'a gpu::SurvivalFlexBlock10TimepointDirectional,
}

/// Entry/exit base + both directional + bidirectional timepoint packs for the
/// contracted-fourth path, bundled to keep `flex_row_nll_fourth_contracted`
/// under the argument-count gate.
pub(crate) struct FlexFourthPacks<'a> {
    pub entry_base: &'a gpu::SurvivalFlexBlock10TimepointBase,
    pub exit_base: &'a gpu::SurvivalFlexBlock10TimepointBase,
    pub entry_ext_u: &'a gpu::SurvivalFlexBlock10TimepointDirectional,
    pub exit_ext_u: &'a gpu::SurvivalFlexBlock10TimepointDirectional,
    pub entry_ext_v: &'a gpu::SurvivalFlexBlock10TimepointDirectional,
    pub exit_ext_v: &'a gpu::SurvivalFlexBlock10TimepointDirectional,
    pub entry_bi: &'a gpu::SurvivalFlexBlock10TimepointBiDirectional,
    pub exit_bi: &'a gpu::SurvivalFlexBlock10TimepointBiDirectional,
}

impl SurvivalMarginalSlopeFamily {
    /// #932 Item 1: dispatch the runtime primary count `p` to a concrete `K` and
    /// lift the calibration intercept Hessian `a_uv` via [`lift_intercept_order2`]
    /// (`filtered_implicit_solve_scalar` over the calibration constraint) — the
    /// single-source replacement for the hand IFT closed form. `Order2` keeps it
    /// `O(K²)` per timepoint (no dense `Tower4<K+1>`); for primary counts beyond
    /// the dispatch table the byte-identical hand IFT is the fallback.
    pub(crate) fn lift_flex_intercept_hessian(
        &self,
        p: usize,
        d_check: f64,
        f_u: &Array1<f64>,
        f_uv: &Array2<f64>,
        f_aa: f64,
        d_u: &Array1<f64>,
        a0: f64,
    ) -> Result<Array2<f64>, String> {
        let fu = f_u
            .as_slice()
            .ok_or_else(|| "intercept lift: f_u must be contiguous".to_string())?;
        let fuv = f_uv
            .as_slice()
            .ok_or_else(|| "intercept lift: f_uv must be contiguous".to_string())?;
        let du = d_u
            .as_slice()
            .ok_or_else(|| "intercept lift: d_u must be contiguous".to_string())?;
        macro_rules! go {
            ($k:literal) => {{
                let a_uv = lift_intercept_order2::<$k>(d_check, fu, fuv, f_aa, du, a0);
                Array2::from_shape_fn((p, p), |(i, j)| a_uv[i][j])
            }};
        }
        let a_uv = match p {
            1 => go!(1),
            2 => go!(2),
            3 => go!(3),
            4 => go!(4),
            5 => go!(5),
            6 => go!(6),
            7 => go!(7),
            8 => go!(8),
            9 => go!(9),
            10 => go!(10),
            11 => go!(11),
            12 => go!(12),
            13 => go!(13),
            14 => go!(14),
            15 => go!(15),
            16 => go!(16),
            17 => go!(17),
            18 => go!(18),
            19 => go!(19),
            20 => go!(20),
            21 => go!(21),
            22 => go!(22),
            23 => go!(23),
            24 => go!(24),
            _ => {
                // Byte-identical hand IFT fallback for primary counts beyond the
                // dispatch table.
                let inv = 1.0 / d_check;
                let mut a_u = Array1::<f64>::zeros(p);
                for u in 0..p {
                    a_u[u] = fu[u] * inv;
                }
                let mut a_uv = Array2::<f64>::zeros((p, p));
                for u in 0..p {
                    for v in u..p {
                        let value = (f_uv[[u, v]]
                            - d_u[u] * a_u[v]
                            - d_u[v] * a_u[u]
                            - f_aa * a_u[u] * a_u[v])
                            * inv;
                        a_uv[[u, v]] = value;
                        a_uv[[v, u]] = value;
                    }
                }
                a_uv
            }
        };
        Ok(a_uv)
    }

    /// Single-source flex row value + gradient (+ Hessian if `hess_h*` non-empty)
    /// from the entry/exit timepoint packs. The Hessian channel is returned only
    /// when the `*_uv` slices are supplied; the grad-only caller passes empty
    /// `h` slices (the value/gradient channels do not read the Hessian).
    ///
    /// `g_*` are the length-`p` gradient packs, `h_*` the `p*p` row-major Hessian
    /// packs (empty for grad-only). Replaces the hand value/grad/Hessian
    /// assembly in `flex_sensitivity.rs`.
    pub(crate) fn flex_row_nll_value_grad_hess(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q1: f64,
        qd1: f64,
        ch: FlexRowJet2Channels<'_>,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let FlexRowJet2Channels {
            eta0_v,
            eta0_g,
            eta0_h,
            eta1_v,
            eta1_g,
            eta1_h,
            chi1_v,
            chi1_g,
            chi1_h,
            d1_v,
            d1_g,
            d1_h,
        } = ch;
        let p = primary.total;
        let wi = self.weights[row];
        let di = self.event[row];
        let surv0 = surv_stack(eta0_v)?;
        let surv1 = surv_stack(eta1_v)?;
        let want_hess = eta1_h.is_some();
        let eta0 = Jet2::from_view(eta0_v, eta0_g, eta0_h);
        let eta1 = Jet2::from_view(eta1_v, eta1_g, eta1_h);
        let chi1 = Jet2::from_view(chi1_v, chi1_g, chi1_h);
        let d1 = Jet2::from_view(d1_v, d1_g, d1_h);
        let q1j = Jet2::primary(q1, primary.q1, p);
        let qd1j = Jet2::primary(qd1, primary.qd1, p);
        let out = flex_row_nll(&eta0, &eta1, &chi1, &d1, &q1j, &qd1j, surv0, surv1, wi, di);
        let value = out.v + wi * di * std::f64::consts::TAU.ln();
        let grad = Array1::from(out.g);
        let hess = if want_hess {
            Array2::from_shape_vec((p, p), out.h).map_err(|e| e.to_string())?
        } else {
            Array2::zeros((p, p))
        };
        Ok((value, grad, hess))
    }

    /// Single-source flex contracted third `D_dir H[u,v]` from the entry/exit
    /// base + directional packs. Replaces `gpu::cpu_oracle_third_contraction`.
    pub(crate) fn flex_row_nll_third_contracted(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q1: f64,
        qd1: f64,
        dir: &[f64],
        packs: FlexThirdPacks<'_>,
    ) -> Result<Array2<f64>, String> {
        let FlexThirdPacks {
            entry_base,
            exit_base,
            entry_ext,
            exit_ext,
        } = packs;
        let p = primary.total;
        let wi = self.weights[row];
        let di = self.event[row];
        let surv0 = surv_stack(entry_base.eta)?;
        let surv1 = surv_stack(exit_base.eta)?;

        let mk = |base_v: f64,
                  base_g: &[f64],
                  base_h: &[f64],
                  ext_g: &[f64],
                  ext_h: &[f64]|
         -> Jet3 {
            Jet3 {
                base: Jet2::from_parts(base_v, base_g, base_h),
                eps: Jet2::from_parts(dot(base_g, dir), ext_g, ext_h),
            }
        };
        let eta0 = mk(
            entry_base.eta,
            &entry_base.eta_u,
            &entry_base.eta_uv,
            &entry_ext.eta_u_dir,
            &entry_ext.eta_uv_dir,
        );
        let eta1 = mk(
            exit_base.eta,
            &exit_base.eta_u,
            &exit_base.eta_uv,
            &exit_ext.eta_u_dir,
            &exit_ext.eta_uv_dir,
        );
        let chi1 = mk(
            exit_base.chi,
            &exit_base.chi_u,
            &exit_base.chi_uv,
            &exit_ext.chi_u_dir,
            &exit_ext.chi_uv_dir,
        );
        let d1 = mk(
            exit_base.d,
            &exit_base.d_u,
            &exit_base.d_uv,
            &exit_ext.d_u_dir,
            &exit_ext.d_uv_dir,
        );
        let q1j = Jet3::primary(q1, primary.q1, p, dir[primary.q1]);
        let qd1j = Jet3::primary(qd1, primary.qd1, p, dir[primary.qd1]);
        let out = flex_row_nll(&eta0, &eta1, &chi1, &d1, &q1j, &qd1j, surv0, surv1, wi, di);
        Array2::from_shape_vec((p, p), out.contracted_third()).map_err(|e| e.to_string())
    }

    /// Single-source flex contracted fourth `Σ_{cd} ℓ_{abcd} u_c v_d` from the
    /// entry/exit base + both directional packs + bidirectional packs. Replaces
    /// `gpu::cpu_oracle_fourth_contraction`.
    pub(crate) fn flex_row_nll_fourth_contracted(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q1: f64,
        qd1: f64,
        dir_u: &[f64],
        dir_v: &[f64],
        packs: FlexFourthPacks<'_>,
    ) -> Result<Array2<f64>, String> {
        let FlexFourthPacks {
            entry_base,
            exit_base,
            entry_ext_u,
            exit_ext_u,
            entry_ext_v,
            exit_ext_v,
            entry_bi,
            exit_bi,
        } = packs;
        let p = primary.total;
        let wi = self.weights[row];
        let di = self.event[row];
        let surv0 = surv_stack(entry_base.eta)?;
        let surv1 = surv_stack(exit_base.eta)?;

        // eps_del.v = uᵀ·H·v, eps_del.g = (H_dir_u)·v, eps_del.h = bi.
        let mk = |base_v: f64,
                  base_g: &[f64],
                  base_h: &[f64],
                  ext_u_g: &[f64],
                  ext_u_h: &[f64],
                  ext_v_g: &[f64],
                  ext_v_h: &[f64],
                  bi_h: &[f64]|
         -> Jet4 {
            let eps_del_v = quad_form(base_h, dir_u, dir_v, p);
            let eps_del_g = mat_vec(ext_u_h, dir_v, p);
            Jet4 {
                base: Jet2::from_parts(base_v, base_g, base_h),
                eps: Jet2::from_parts(dot(base_g, dir_u), ext_u_g, ext_u_h),
                del: Jet2::from_parts(dot(base_g, dir_v), ext_v_g, ext_v_h),
                eps_del: Jet2::from_parts(eps_del_v, &eps_del_g, bi_h),
            }
        };
        let eta0 = mk(
            entry_base.eta,
            &entry_base.eta_u,
            &entry_base.eta_uv,
            &entry_ext_u.eta_u_dir,
            &entry_ext_u.eta_uv_dir,
            &entry_ext_v.eta_u_dir,
            &entry_ext_v.eta_uv_dir,
            &entry_bi.eta_uv_uv,
        );
        let eta1 = mk(
            exit_base.eta,
            &exit_base.eta_u,
            &exit_base.eta_uv,
            &exit_ext_u.eta_u_dir,
            &exit_ext_u.eta_uv_dir,
            &exit_ext_v.eta_u_dir,
            &exit_ext_v.eta_uv_dir,
            &exit_bi.eta_uv_uv,
        );
        let chi1 = mk(
            exit_base.chi,
            &exit_base.chi_u,
            &exit_base.chi_uv,
            &exit_ext_u.chi_u_dir,
            &exit_ext_u.chi_uv_dir,
            &exit_ext_v.chi_u_dir,
            &exit_ext_v.chi_uv_dir,
            &exit_bi.chi_uv_uv,
        );
        let d1 = mk(
            exit_base.d,
            &exit_base.d_u,
            &exit_base.d_uv,
            &exit_ext_u.d_u_dir,
            &exit_ext_u.d_uv_dir,
            &exit_ext_v.d_u_dir,
            &exit_ext_v.d_uv_dir,
            &exit_bi.d_uv_uv,
        );
        let q1j = Jet4::primary(q1, primary.q1, p, dir_u[primary.q1], dir_v[primary.q1]);
        let qd1j = Jet4::primary(qd1, primary.qd1, p, dir_u[primary.qd1], dir_v[primary.qd1]);
        let out = flex_row_nll(&eta0, &eta1, &chi1, &d1, &q1j, &qd1j, surv0, surv1, wi, di);
        Array2::from_shape_vec((p, p), out.contracted_fourth()).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod moment_engine_tests {
    use super::*;
    use crate::families::cubic_cell_kernel::{reduce_sextic_moments, DenestedCubicCell};

    /// #932 item-2 increment 1: the FlexJet moment recurrence must reproduce the
    /// numeric `reduce_sextic_moments` on the VALUE channel term-for-term (a
    /// generic non-degenerate sextic cell), proving the port of the raising
    /// recurrence + boundary term to the jet algebra is exact. (Derivative
    /// channels are exercised by the full timepoint oracle once Phase C lands.)
    #[test]
    fn cell_moment_recurrence_jet_value_matches_numeric_932() {
        let cell = DenestedCubicCell {
            left: -1.5,
            right: 2.0,
            c0: 0.3,
            c1: -0.4,
            c2: 0.5,
            c3: 0.2,
        };
        let base = [1.0_f64, 0.1, 0.6, -0.05, 0.4];
        let max_degree = 12usize;
        let reference =
            reduce_sextic_moments(cell, base, max_degree).expect("numeric sextic moments");

        let p = 3usize;
        let konst = |x: f64| Jet2::from_parts(x, &vec![0.0; p], &[]);
        let c = [
            konst(cell.c0),
            konst(cell.c1),
            konst(cell.c2),
            konst(cell.c3),
        ];
        let zl = konst(cell.left);
        let zr = konst(cell.right);
        let base_jets = [
            konst(base[0]),
            konst(base[1]),
            konst(base[2]),
            konst(base[3]),
            konst(base[4]),
        ];
        let moments = cell_moment_recurrence_jet(
            &c,
            &zl,
            cell.left.is_finite(),
            &zr,
            cell.right.is_finite(),
            &base_jets,
            max_degree,
        );
        assert_eq!(moments.len(), reference.len(), "moment count");
        for (n, (m, r)) in moments.iter().zip(reference.iter()).enumerate() {
            assert!(
                (m.value() - r).abs() <= 1e-9 * (1.0 + r.abs()),
                "moment {n}: jet value {} != numeric {}",
                m.value(),
                r
            );
        }
    }
}
